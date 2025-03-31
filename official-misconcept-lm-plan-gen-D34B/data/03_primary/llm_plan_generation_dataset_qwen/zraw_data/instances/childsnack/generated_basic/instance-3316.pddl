; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 338213

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child52 child380 - child
    bread285 bread471 - bread-portion
    content413 content59 - content-portion
    tray33 tray249 tray138 tray258 - tray
    table357 table335 table367 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray33 kitchen)
     (at tray249 kitchen)
     (at tray138 kitchen)
     (at tray258 kitchen)
     (at_kitchen_bread bread285)
     (at_kitchen_bread bread471)
     (at_kitchen_content content413)
     (at_kitchen_content content59)
     (no_gluten_bread bread471)
     (no_gluten_content content59)
     (allergic_gluten child380)
     (not_allergic_gluten child52)
     (waiting child52 table367)
     (waiting child380 table335)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child52)
     (served child380)
    )
  )
)
