; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 621181

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child337 - child
    bread283 - bread-portion
    content286 - content-portion
    tray192 tray447 tray218 tray332 - tray
    table358 table390 table78 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray192 kitchen)
     (at tray447 kitchen)
     (at tray218 kitchen)
     (at tray332 kitchen)
     (at_kitchen_bread bread283)
     (at_kitchen_content content286)
     (not_allergic_gluten child337)
     (waiting child337 table78)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child337)
    )
  )
)
