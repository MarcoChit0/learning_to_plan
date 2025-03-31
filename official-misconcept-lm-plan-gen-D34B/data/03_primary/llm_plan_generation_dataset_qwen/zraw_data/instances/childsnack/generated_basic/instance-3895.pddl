; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 680277

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child93 - child
    bread374 - bread-portion
    content291 - content-portion
    tray272 tray241 - tray
    table212 table69 table130 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray272 kitchen)
     (at tray241 kitchen)
     (at_kitchen_bread bread374)
     (at_kitchen_content content291)
     (not_allergic_gluten child93)
     (waiting child93 table212)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child93)
    )
  )
)
