; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 452412

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child21 - child
    bread221 - bread-portion
    content358 - content-portion
    tray474 - tray
    table296 table278 table123 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray474 kitchen)
     (at_kitchen_bread bread221)
     (at_kitchen_content content358)
     (not_allergic_gluten child21)
     (waiting child21 table296)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child21)
    )
  )
)
