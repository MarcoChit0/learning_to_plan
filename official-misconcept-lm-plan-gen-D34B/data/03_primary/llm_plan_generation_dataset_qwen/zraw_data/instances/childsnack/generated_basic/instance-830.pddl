; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 248424

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child469 child305 - child
    bread471 bread490 - bread-portion
    content355 content250 - content-portion
    tray66 tray30 - tray
    table197 table14 table286 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray66 kitchen)
     (at tray30 kitchen)
     (at_kitchen_bread bread471)
     (at_kitchen_bread bread490)
     (at_kitchen_content content355)
     (at_kitchen_content content250)
     (no_gluten_bread bread471)
     (no_gluten_content content250)
     (allergic_gluten child305)
     (not_allergic_gluten child469)
     (waiting child469 table286)
     (waiting child305 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child469)
     (served child305)
    )
  )
)
