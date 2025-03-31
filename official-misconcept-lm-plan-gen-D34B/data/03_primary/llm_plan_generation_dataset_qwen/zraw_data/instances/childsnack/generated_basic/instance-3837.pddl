; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 480014

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child414 - child
    bread255 - bread-portion
    content90 - content-portion
    tray441 tray455 tray124 tray116 - tray
    table416 table71 table322 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray441 kitchen)
     (at tray455 kitchen)
     (at tray124 kitchen)
     (at tray116 kitchen)
     (at_kitchen_bread bread255)
     (at_kitchen_content content90)
     (not_allergic_gluten child414)
     (waiting child414 table71)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child414)
    )
  )
)
