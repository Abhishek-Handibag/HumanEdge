import React, { useState } from 'react';
import './login.css';

const Login = () => {
    const [loginStage, setLoginStage] = useState('email'); // email, otp
    const [email, setEmail] = useState('');
    const [otp, setOtp] = useState('');

    const handleEmailSubmit = (e) => {
        e.preventDefault();
        // Here you would typically trigger OTP send to email
        setLoginStage('otp');
    };

    const handleOtpSubmit = (e) => {
        e.preventDefault();
        // Here you would verify OTP
        console.log('OTP Verification');
    };

    return (
        <div className="login-container">
            <div className="login-box">
                <div className="login-header">
                    <h2>Welcome Back</h2>
                    <p>Login to continue</p>
                </div>

                {loginStage === 'email' ? (
                    <form onSubmit={handleEmailSubmit} className="login-form">
                        <div className="form-group">
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="Enter your email"
                                required
                            />
                            <span className="focus-border"></span>
                        </div>
                        <button type="submit" className="login-button">
                            Get OTP
                            <span className="button-effect"></span>
                        </button>
                    </form>
                ) : (
                    <form onSubmit={handleOtpSubmit} className="login-form">
                        <div className="form-group">
                            <input
                                type="text"
                                value={otp}
                                onChange={(e) => setOtp(e.target.value)}
                                placeholder="Enter OTP"
                                maxLength="6"
                                required
                            />
                            <span className="focus-border"></span>
                        </div>
                        <button type="submit" className="login-button">
                            Verify OTP
                            <span className="button-effect"></span>
                        </button>
                        <button 
                            type="button" 
                            className="back-button"
                            onClick={() => setLoginStage('email')}
                        >
                            Back to Email
                        </button>
                    </form>
                )}
            </div>
        </div>
    );
};

export default Login; 