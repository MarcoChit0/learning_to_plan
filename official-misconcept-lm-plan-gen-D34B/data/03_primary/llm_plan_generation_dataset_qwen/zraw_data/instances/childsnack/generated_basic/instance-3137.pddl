; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 787997

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child344 child390 - child
    bread184 bread174 - bread-portion
    content361 content205 - content-portion
    tray103 tray232 - tray
    table157 table485 table360 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray103 kitchen)
     (at tray232 kitchen)
     (at_kitchen_bread bread184)
     (at_kitchen_bread bread174)
     (at_kitchen_content content361)
     (at_kitchen_content content205)
     (not_allergic_gluten child344)
     (not_allergic_gluten child390)
     (waiting child344 table157)
     (waiting child390 table485)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child344)
     (served child390)
    )
  )
)
