; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 842847

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child39 child381 - child
    bread389 bread432 - bread-portion
    content212 content432 - content-portion
    tray214 tray44 - tray
    table390 table198 table307 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray214 kitchen)
     (at tray44 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_bread bread432)
     (at_kitchen_content content212)
     (at_kitchen_content content432)
     (no_gluten_bread bread432)
     (no_gluten_content content212)
     (allergic_gluten child381)
     (not_allergic_gluten child39)
     (waiting child39 table307)
     (waiting child381 table198)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child39)
     (served child381)
    )
  )
)
