; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 526876

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child103 child243 - child
    bread255 bread182 - bread-portion
    content110 content380 - content-portion
    tray351 tray188 - tray
    table230 table396 table242 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray351 kitchen)
     (at tray188 kitchen)
     (at_kitchen_bread bread255)
     (at_kitchen_bread bread182)
     (at_kitchen_content content110)
     (at_kitchen_content content380)
     (not_allergic_gluten child103)
     (not_allergic_gluten child243)
     (waiting child103 table230)
     (waiting child243 table396)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child103)
     (served child243)
    )
  )
)
