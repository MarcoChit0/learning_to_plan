; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 731070

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child45 - child
    bread177 - bread-portion
    content280 - content-portion
    tray384 tray452 - tray
    table371 table55 table487 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at tray452 kitchen)
     (at_kitchen_bread bread177)
     (at_kitchen_content content280)
     (not_allergic_gluten child45)
     (waiting child45 table55)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child45)
    )
  )
)
