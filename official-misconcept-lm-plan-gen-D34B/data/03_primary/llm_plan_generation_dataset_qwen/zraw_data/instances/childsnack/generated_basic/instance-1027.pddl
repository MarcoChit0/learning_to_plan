; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 364538

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child441 child450 child326 - child
    bread321 bread391 bread37 - bread-portion
    content81 content161 content187 - content-portion
    tray128 - tray
    table496 table162 table375 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread391)
     (at_kitchen_bread bread37)
     (at_kitchen_content content81)
     (at_kitchen_content content161)
     (at_kitchen_content content187)
     (not_allergic_gluten child441)
     (not_allergic_gluten child326)
     (not_allergic_gluten child450)
     (waiting child441 table375)
     (waiting child450 table162)
     (waiting child326 table375)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child441)
     (served child450)
     (served child326)
    )
  )
)
