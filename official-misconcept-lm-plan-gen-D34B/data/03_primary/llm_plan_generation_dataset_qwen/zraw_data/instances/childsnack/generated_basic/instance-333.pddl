; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 186012

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child304 - child
    bread52 - bread-portion
    content122 - content-portion
    tray131 tray59 tray33 tray332 - tray
    table363 table288 table132 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray131 kitchen)
     (at tray59 kitchen)
     (at tray33 kitchen)
     (at tray332 kitchen)
     (at_kitchen_bread bread52)
     (at_kitchen_content content122)
     (not_allergic_gluten child304)
     (waiting child304 table132)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child304)
    )
  )
)
