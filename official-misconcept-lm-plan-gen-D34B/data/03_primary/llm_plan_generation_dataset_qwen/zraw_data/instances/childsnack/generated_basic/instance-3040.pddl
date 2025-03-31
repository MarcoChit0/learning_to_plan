; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 753701

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child52 child101 - child
    bread316 bread134 - bread-portion
    content259 content488 - content-portion
    tray14 tray325 tray83 - tray
    table271 table321 table184 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray14 kitchen)
     (at tray325 kitchen)
     (at tray83 kitchen)
     (at_kitchen_bread bread316)
     (at_kitchen_bread bread134)
     (at_kitchen_content content259)
     (at_kitchen_content content488)
     (no_gluten_bread bread316)
     (no_gluten_content content488)
     (allergic_gluten child101)
     (not_allergic_gluten child52)
     (waiting child52 table321)
     (waiting child101 table184)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child52)
     (served child101)
    )
  )
)
