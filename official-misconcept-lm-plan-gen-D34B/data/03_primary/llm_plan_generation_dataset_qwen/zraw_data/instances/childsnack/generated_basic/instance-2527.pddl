; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 713312

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child57 - child
    bread299 - bread-portion
    content116 - content-portion
    tray362 tray155 tray234 - tray
    table290 table335 table296 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray362 kitchen)
     (at tray155 kitchen)
     (at tray234 kitchen)
     (at_kitchen_bread bread299)
     (at_kitchen_content content116)
     (not_allergic_gluten child57)
     (waiting child57 table335)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child57)
    )
  )
)
