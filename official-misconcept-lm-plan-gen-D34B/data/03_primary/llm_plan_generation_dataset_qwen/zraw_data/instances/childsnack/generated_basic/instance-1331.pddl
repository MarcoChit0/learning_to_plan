; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 716338

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child480 child269 - child
    bread327 bread439 - bread-portion
    content24 content26 - content-portion
    tray451 tray409 tray312 - tray
    table186 table284 table446 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray451 kitchen)
     (at tray409 kitchen)
     (at tray312 kitchen)
     (at_kitchen_bread bread327)
     (at_kitchen_bread bread439)
     (at_kitchen_content content24)
     (at_kitchen_content content26)
     (not_allergic_gluten child269)
     (not_allergic_gluten child480)
     (waiting child480 table186)
     (waiting child269 table446)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child480)
     (served child269)
    )
  )
)
