; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 924561

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 - child
    bread266 - bread-portion
    content170 - content-portion
    tray82 tray61 tray244 - tray
    table240 table317 table288 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray82 kitchen)
     (at tray61 kitchen)
     (at tray244 kitchen)
     (at_kitchen_bread bread266)
     (at_kitchen_content content170)
     (not_allergic_gluten child200)
     (waiting child200 table240)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child200)
    )
  )
)
