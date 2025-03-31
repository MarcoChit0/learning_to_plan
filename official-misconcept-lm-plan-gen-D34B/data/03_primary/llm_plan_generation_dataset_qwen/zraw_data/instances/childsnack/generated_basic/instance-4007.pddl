; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 655915

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child215 - child
    bread490 - bread-portion
    content437 - content-portion
    tray249 tray318 tray291 - tray
    table3 table132 table346 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray249 kitchen)
     (at tray318 kitchen)
     (at tray291 kitchen)
     (at_kitchen_bread bread490)
     (at_kitchen_content content437)
     (not_allergic_gluten child215)
     (waiting child215 table346)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child215)
    )
  )
)
