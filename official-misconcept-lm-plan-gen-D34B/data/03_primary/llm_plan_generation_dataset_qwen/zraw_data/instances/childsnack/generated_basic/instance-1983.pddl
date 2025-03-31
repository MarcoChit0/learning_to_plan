; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 388080

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child467 child243 - child
    bread428 bread55 - bread-portion
    content103 content168 - content-portion
    tray84 - tray
    table371 table307 table300 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray84 kitchen)
     (at_kitchen_bread bread428)
     (at_kitchen_bread bread55)
     (at_kitchen_content content103)
     (at_kitchen_content content168)
     (no_gluten_bread bread428)
     (no_gluten_content content103)
     (allergic_gluten child243)
     (not_allergic_gluten child467)
     (waiting child467 table371)
     (waiting child243 table300)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child467)
     (served child243)
    )
  )
)
