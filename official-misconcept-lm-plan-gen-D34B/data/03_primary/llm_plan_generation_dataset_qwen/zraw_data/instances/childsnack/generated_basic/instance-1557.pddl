; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 587517

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child155 child246 - child
    bread165 bread268 - bread-portion
    content462 content8 - content-portion
    tray94 tray252 tray353 tray437 - tray
    table158 table339 table399 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray94 kitchen)
     (at tray252 kitchen)
     (at tray353 kitchen)
     (at tray437 kitchen)
     (at_kitchen_bread bread165)
     (at_kitchen_bread bread268)
     (at_kitchen_content content462)
     (at_kitchen_content content8)
     (not_allergic_gluten child246)
     (not_allergic_gluten child155)
     (waiting child155 table158)
     (waiting child246 table399)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child155)
     (served child246)
    )
  )
)
