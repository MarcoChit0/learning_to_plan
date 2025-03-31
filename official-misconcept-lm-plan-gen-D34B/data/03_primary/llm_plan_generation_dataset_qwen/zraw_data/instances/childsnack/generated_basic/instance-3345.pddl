; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 217752

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child104 child496 - child
    bread61 bread399 - bread-portion
    content476 content28 - content-portion
    tray218 tray83 - tray
    table316 table127 table188 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray218 kitchen)
     (at tray83 kitchen)
     (at_kitchen_bread bread61)
     (at_kitchen_bread bread399)
     (at_kitchen_content content476)
     (at_kitchen_content content28)
     (no_gluten_bread bread61)
     (no_gluten_content content28)
     (allergic_gluten child104)
     (not_allergic_gluten child496)
     (waiting child104 table188)
     (waiting child496 table316)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child104)
     (served child496)
    )
  )
)
