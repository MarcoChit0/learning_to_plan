; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 448638

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child145 - child
    bread83 - bread-portion
    content473 - content-portion
    tray342 tray98 tray419 tray329 - tray
    table237 table480 table290 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray342 kitchen)
     (at tray98 kitchen)
     (at tray419 kitchen)
     (at tray329 kitchen)
     (at_kitchen_bread bread83)
     (at_kitchen_content content473)
     (not_allergic_gluten child145)
     (waiting child145 table237)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child145)
    )
  )
)
