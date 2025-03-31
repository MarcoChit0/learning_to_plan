; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 618391

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child100 child391 - child
    bread213 bread252 - bread-portion
    content38 content442 - content-portion
    tray351 - tray
    table192 table208 table28 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray351 kitchen)
     (at_kitchen_bread bread213)
     (at_kitchen_bread bread252)
     (at_kitchen_content content38)
     (at_kitchen_content content442)
     (no_gluten_bread bread213)
     (no_gluten_content content442)
     (allergic_gluten child100)
     (not_allergic_gluten child391)
     (waiting child100 table192)
     (waiting child391 table192)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child100)
     (served child391)
    )
  )
)
