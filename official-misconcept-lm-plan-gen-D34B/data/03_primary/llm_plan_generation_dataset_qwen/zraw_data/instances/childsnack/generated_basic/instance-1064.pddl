; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 59901

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child322 child82 - child
    bread120 bread109 - bread-portion
    content20 content431 - content-portion
    tray343 tray249 tray402 - tray
    table356 table465 table289 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray343 kitchen)
     (at tray249 kitchen)
     (at tray402 kitchen)
     (at_kitchen_bread bread120)
     (at_kitchen_bread bread109)
     (at_kitchen_content content20)
     (at_kitchen_content content431)
     (not_allergic_gluten child322)
     (not_allergic_gluten child82)
     (waiting child322 table289)
     (waiting child82 table465)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child322)
     (served child82)
    )
  )
)
