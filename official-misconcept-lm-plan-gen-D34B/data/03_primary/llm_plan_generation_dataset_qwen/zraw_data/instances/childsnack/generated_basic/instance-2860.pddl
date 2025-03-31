; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 18889

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child440 - child
    bread67 - bread-portion
    content202 - content-portion
    tray220 tray11 - tray
    table51 table193 table461 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray220 kitchen)
     (at tray11 kitchen)
     (at_kitchen_bread bread67)
     (at_kitchen_content content202)
     (not_allergic_gluten child440)
     (waiting child440 table51)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child440)
    )
  )
)
