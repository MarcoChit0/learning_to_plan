; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 979887

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child199 child355 - child
    bread373 bread364 - bread-portion
    content379 content197 - content-portion
    tray418 tray56 tray212 tray253 - tray
    table358 table194 table339 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at tray56 kitchen)
     (at tray212 kitchen)
     (at tray253 kitchen)
     (at_kitchen_bread bread373)
     (at_kitchen_bread bread364)
     (at_kitchen_content content379)
     (at_kitchen_content content197)
     (not_allergic_gluten child199)
     (not_allergic_gluten child355)
     (waiting child199 table358)
     (waiting child355 table194)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child199)
     (served child355)
    )
  )
)
