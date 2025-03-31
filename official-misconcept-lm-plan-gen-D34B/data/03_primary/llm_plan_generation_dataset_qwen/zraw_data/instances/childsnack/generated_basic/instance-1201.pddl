; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 651743

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child437 child443 child147 - child
    bread436 bread92 bread118 - bread-portion
    content418 content2 content12 - content-portion
    tray304 tray32 tray391 - tray
    table303 table392 table442 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray304 kitchen)
     (at tray32 kitchen)
     (at tray391 kitchen)
     (at_kitchen_bread bread436)
     (at_kitchen_bread bread92)
     (at_kitchen_bread bread118)
     (at_kitchen_content content418)
     (at_kitchen_content content2)
     (at_kitchen_content content12)
     (not_allergic_gluten child437)
     (not_allergic_gluten child147)
     (not_allergic_gluten child443)
     (waiting child437 table303)
     (waiting child443 table303)
     (waiting child147 table442)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child437)
     (served child443)
     (served child147)
    )
  )
)
