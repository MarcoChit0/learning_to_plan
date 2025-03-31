; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 745449

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child66 child125 child103 - child
    bread38 bread239 bread200 - bread-portion
    content413 content304 content420 - content-portion
    tray452 tray260 tray214 - tray
    table161 table471 table33 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray452 kitchen)
     (at tray260 kitchen)
     (at tray214 kitchen)
     (at_kitchen_bread bread38)
     (at_kitchen_bread bread239)
     (at_kitchen_bread bread200)
     (at_kitchen_content content413)
     (at_kitchen_content content304)
     (at_kitchen_content content420)
     (not_allergic_gluten child103)
     (not_allergic_gluten child125)
     (not_allergic_gluten child66)
     (waiting child66 table471)
     (waiting child125 table471)
     (waiting child103 table161)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child66)
     (served child125)
     (served child103)
    )
  )
)
