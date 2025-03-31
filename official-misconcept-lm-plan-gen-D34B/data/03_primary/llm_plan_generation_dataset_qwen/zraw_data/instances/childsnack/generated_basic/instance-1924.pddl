; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 874101

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child167 - child
    bread389 - bread-portion
    content150 - content-portion
    tray7 tray44 tray246 - tray
    table70 table84 table331 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray7 kitchen)
     (at tray44 kitchen)
     (at tray246 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_content content150)
     (not_allergic_gluten child167)
     (waiting child167 table84)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child167)
    )
  )
)
