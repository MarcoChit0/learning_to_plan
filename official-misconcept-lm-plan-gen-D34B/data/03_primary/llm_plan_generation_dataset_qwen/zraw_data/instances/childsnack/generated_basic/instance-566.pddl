; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 88183

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child115 child339 - child
    bread99 bread15 - bread-portion
    content283 content391 - content-portion
    tray478 - tray
    table326 table134 table89 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray478 kitchen)
     (at_kitchen_bread bread99)
     (at_kitchen_bread bread15)
     (at_kitchen_content content283)
     (at_kitchen_content content391)
     (no_gluten_bread bread99)
     (no_gluten_content content283)
     (allergic_gluten child339)
     (not_allergic_gluten child115)
     (waiting child115 table89)
     (waiting child339 table89)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child115)
     (served child339)
    )
  )
)
