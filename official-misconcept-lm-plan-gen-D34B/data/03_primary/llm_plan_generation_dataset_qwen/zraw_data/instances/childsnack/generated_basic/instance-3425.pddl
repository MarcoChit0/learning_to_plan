; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 855405

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child321 - child
    bread336 - bread-portion
    content233 - content-portion
    tray18 tray297 tray24 tray86 - tray
    table78 table238 table218 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at tray297 kitchen)
     (at tray24 kitchen)
     (at tray86 kitchen)
     (at_kitchen_bread bread336)
     (at_kitchen_content content233)
     (not_allergic_gluten child321)
     (waiting child321 table78)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child321)
    )
  )
)
