; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 557670

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child286 child380 - child
    bread305 bread262 - bread-portion
    content432 content419 - content-portion
    tray60 tray51 - tray
    table469 table55 table116 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray60 kitchen)
     (at tray51 kitchen)
     (at_kitchen_bread bread305)
     (at_kitchen_bread bread262)
     (at_kitchen_content content432)
     (at_kitchen_content content419)
     (no_gluten_bread bread262)
     (no_gluten_content content419)
     (allergic_gluten child286)
     (not_allergic_gluten child380)
     (waiting child286 table55)
     (waiting child380 table469)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child286)
     (served child380)
    )
  )
)
