; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 613147

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 child194 - child
    bread281 bread83 - bread-portion
    content98 content202 - content-portion
    tray375 tray37 tray179 - tray
    table217 table241 table244 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray375 kitchen)
     (at tray37 kitchen)
     (at tray179 kitchen)
     (at_kitchen_bread bread281)
     (at_kitchen_bread bread83)
     (at_kitchen_content content98)
     (at_kitchen_content content202)
     (not_allergic_gluten child73)
     (not_allergic_gluten child194)
     (waiting child73 table241)
     (waiting child194 table241)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child73)
     (served child194)
    )
  )
)
