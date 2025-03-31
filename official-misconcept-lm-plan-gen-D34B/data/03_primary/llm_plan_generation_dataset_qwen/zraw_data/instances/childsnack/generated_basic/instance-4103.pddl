; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 547112

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child295 - child
    bread471 - bread-portion
    content109 - content-portion
    tray3 tray457 tray430 - tray
    table441 table138 table345 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray3 kitchen)
     (at tray457 kitchen)
     (at tray430 kitchen)
     (at_kitchen_bread bread471)
     (at_kitchen_content content109)
     (not_allergic_gluten child295)
     (waiting child295 table138)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child295)
    )
  )
)
