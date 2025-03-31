; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 188107

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child498 - child
    bread102 - bread-portion
    content245 - content-portion
    tray327 tray360 tray331 tray455 - tray
    table283 table466 table220 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray327 kitchen)
     (at tray360 kitchen)
     (at tray331 kitchen)
     (at tray455 kitchen)
     (at_kitchen_bread bread102)
     (at_kitchen_content content245)
     (not_allergic_gluten child498)
     (waiting child498 table220)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child498)
    )
  )
)
