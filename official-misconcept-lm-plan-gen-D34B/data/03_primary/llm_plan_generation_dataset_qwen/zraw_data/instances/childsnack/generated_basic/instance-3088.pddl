; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 7804

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child389 child420 - child
    bread476 bread344 - bread-portion
    content444 content168 - content-portion
    tray166 tray96 tray253 - tray
    table275 table12 table304 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray166 kitchen)
     (at tray96 kitchen)
     (at tray253 kitchen)
     (at_kitchen_bread bread476)
     (at_kitchen_bread bread344)
     (at_kitchen_content content444)
     (at_kitchen_content content168)
     (not_allergic_gluten child389)
     (not_allergic_gluten child420)
     (waiting child389 table12)
     (waiting child420 table304)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child389)
     (served child420)
    )
  )
)
