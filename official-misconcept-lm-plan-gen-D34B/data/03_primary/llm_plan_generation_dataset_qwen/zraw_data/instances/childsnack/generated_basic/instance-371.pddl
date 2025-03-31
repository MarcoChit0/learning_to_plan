; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 682227

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child388 - child
    bread131 - bread-portion
    content189 - content-portion
    tray343 tray368 tray395 tray128 - tray
    table27 table336 table433 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray343 kitchen)
     (at tray368 kitchen)
     (at tray395 kitchen)
     (at tray128 kitchen)
     (at_kitchen_bread bread131)
     (at_kitchen_content content189)
     (not_allergic_gluten child388)
     (waiting child388 table27)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child388)
    )
  )
)
