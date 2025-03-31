; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 595934

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child322 - child
    bread432 - bread-portion
    content367 - content-portion
    tray371 tray253 tray65 tray183 - tray
    table130 table4 table93 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray371 kitchen)
     (at tray253 kitchen)
     (at tray65 kitchen)
     (at tray183 kitchen)
     (at_kitchen_bread bread432)
     (at_kitchen_content content367)
     (not_allergic_gluten child322)
     (waiting child322 table130)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child322)
    )
  )
)
