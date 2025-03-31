; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 801431

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child460 - child
    bread412 - bread-portion
    content293 - content-portion
    tray100 - tray
    table343 table360 table36 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray100 kitchen)
     (at_kitchen_bread bread412)
     (at_kitchen_content content293)
     (not_allergic_gluten child460)
     (waiting child460 table36)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child460)
    )
  )
)
