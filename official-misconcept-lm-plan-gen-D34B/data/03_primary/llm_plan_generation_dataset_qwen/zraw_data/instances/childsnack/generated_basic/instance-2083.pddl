; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 187880

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child474 - child
    bread304 - bread-portion
    content280 - content-portion
    tray291 tray152 tray218 tray271 - tray
    table163 table71 table422 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray291 kitchen)
     (at tray152 kitchen)
     (at tray218 kitchen)
     (at tray271 kitchen)
     (at_kitchen_bread bread304)
     (at_kitchen_content content280)
     (not_allergic_gluten child474)
     (waiting child474 table71)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child474)
    )
  )
)
