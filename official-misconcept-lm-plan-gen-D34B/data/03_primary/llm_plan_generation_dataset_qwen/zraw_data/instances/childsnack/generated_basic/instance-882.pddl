; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 284217

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child338 child408 child396 - child
    bread245 bread471 bread273 - bread-portion
    content13 content335 content210 - content-portion
    tray45 - tray
    table351 table105 table208 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray45 kitchen)
     (at_kitchen_bread bread245)
     (at_kitchen_bread bread471)
     (at_kitchen_bread bread273)
     (at_kitchen_content content13)
     (at_kitchen_content content335)
     (at_kitchen_content content210)
     (not_allergic_gluten child396)
     (not_allergic_gluten child338)
     (not_allergic_gluten child408)
     (waiting child338 table351)
     (waiting child408 table105)
     (waiting child396 table351)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child338)
     (served child408)
     (served child396)
    )
  )
)
