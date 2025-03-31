; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 602563

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 - child
    bread58 - bread-portion
    content285 - content-portion
    tray380 tray448 tray56 tray32 - tray
    table400 table447 table14 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray380 kitchen)
     (at tray448 kitchen)
     (at tray56 kitchen)
     (at tray32 kitchen)
     (at_kitchen_bread bread58)
     (at_kitchen_content content285)
     (not_allergic_gluten child163)
     (waiting child163 table14)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child163)
    )
  )
)
