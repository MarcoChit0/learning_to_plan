; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 580759

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child406 - child
    bread92 - bread-portion
    content244 - content-portion
    tray131 tray205 tray95 tray338 - tray
    table358 table227 table60 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray131 kitchen)
     (at tray205 kitchen)
     (at tray95 kitchen)
     (at tray338 kitchen)
     (at_kitchen_bread bread92)
     (at_kitchen_content content244)
     (not_allergic_gluten child406)
     (waiting child406 table358)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child406)
    )
  )
)
