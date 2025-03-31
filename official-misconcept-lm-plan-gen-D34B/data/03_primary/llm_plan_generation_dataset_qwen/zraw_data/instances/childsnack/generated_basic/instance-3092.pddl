; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 341053

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child333 child466 - child
    bread261 bread390 - bread-portion
    content28 content299 - content-portion
    tray48 tray444 tray95 - tray
    table86 table492 table335 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray48 kitchen)
     (at tray444 kitchen)
     (at tray95 kitchen)
     (at_kitchen_bread bread261)
     (at_kitchen_bread bread390)
     (at_kitchen_content content28)
     (at_kitchen_content content299)
     (not_allergic_gluten child333)
     (not_allergic_gluten child466)
     (waiting child333 table492)
     (waiting child466 table86)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child333)
     (served child466)
    )
  )
)
