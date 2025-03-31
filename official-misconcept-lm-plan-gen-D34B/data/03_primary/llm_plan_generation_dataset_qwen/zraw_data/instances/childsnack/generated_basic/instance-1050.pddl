; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 206826

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 child397 - child
    bread477 bread434 - bread-portion
    content400 content311 - content-portion
    tray359 tray112 tray54 - tray
    table450 table419 table401 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray359 kitchen)
     (at tray112 kitchen)
     (at tray54 kitchen)
     (at_kitchen_bread bread477)
     (at_kitchen_bread bread434)
     (at_kitchen_content content400)
     (at_kitchen_content content311)
     (not_allergic_gluten child397)
     (not_allergic_gluten child200)
     (waiting child200 table419)
     (waiting child397 table450)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child200)
     (served child397)
    )
  )
)
