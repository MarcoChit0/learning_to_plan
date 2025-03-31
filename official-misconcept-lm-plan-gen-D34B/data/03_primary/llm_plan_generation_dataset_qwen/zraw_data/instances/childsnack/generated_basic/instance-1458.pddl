; child-snack task with 3 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 972448

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 child339 child345 - child
    bread411 bread28 bread469 - bread-portion
    content31 content478 content419 - content-portion
    tray10 - tray
    table357 table19 table340 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray10 kitchen)
     (at_kitchen_bread bread411)
     (at_kitchen_bread bread28)
     (at_kitchen_bread bread469)
     (at_kitchen_content content31)
     (at_kitchen_content content478)
     (at_kitchen_content content419)
     (no_gluten_bread bread469)
     (no_gluten_bread bread28)
     (no_gluten_content content478)
     (no_gluten_content content419)
     (allergic_gluten child345)
     (allergic_gluten child339)
     (not_allergic_gluten child396)
     (waiting child396 table340)
     (waiting child339 table340)
     (waiting child345 table19)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child396)
     (served child339)
     (served child345)
    )
  )
)
