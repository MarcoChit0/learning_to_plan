; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 767297

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 - child
    bread266 - bread-portion
    content233 - content-portion
    tray33 tray384 tray241 - tray
    table47 table451 table466 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray33 kitchen)
     (at tray384 kitchen)
     (at tray241 kitchen)
     (at_kitchen_bread bread266)
     (at_kitchen_content content233)
     (not_allergic_gluten child92)
     (waiting child92 table451)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child92)
    )
  )
)
