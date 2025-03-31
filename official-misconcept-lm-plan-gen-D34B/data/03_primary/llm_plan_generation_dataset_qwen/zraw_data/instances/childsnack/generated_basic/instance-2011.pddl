; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 479752

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child475 child288 - child
    bread180 bread464 - bread-portion
    content430 content163 - content-portion
    tray61 tray459 tray110 - tray
    table426 table144 table70 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray61 kitchen)
     (at tray459 kitchen)
     (at tray110 kitchen)
     (at_kitchen_bread bread180)
     (at_kitchen_bread bread464)
     (at_kitchen_content content430)
     (at_kitchen_content content163)
     (not_allergic_gluten child475)
     (not_allergic_gluten child288)
     (waiting child475 table70)
     (waiting child288 table70)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child475)
     (served child288)
    )
  )
)
