; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 526881

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child477 child425 - child
    bread413 bread331 - bread-portion
    content429 content443 - content-portion
    tray318 tray447 - tray
    table304 table250 table61 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray318 kitchen)
     (at tray447 kitchen)
     (at_kitchen_bread bread413)
     (at_kitchen_bread bread331)
     (at_kitchen_content content429)
     (at_kitchen_content content443)
     (not_allergic_gluten child425)
     (not_allergic_gluten child477)
     (waiting child477 table304)
     (waiting child425 table250)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child477)
     (served child425)
    )
  )
)
