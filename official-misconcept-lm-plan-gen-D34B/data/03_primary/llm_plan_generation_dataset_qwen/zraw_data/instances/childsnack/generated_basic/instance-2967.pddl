; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 710507

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child255 - child
    bread378 - bread-portion
    content343 - content-portion
    tray450 tray382 tray239 tray328 - tray
    table299 table244 table383 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray450 kitchen)
     (at tray382 kitchen)
     (at tray239 kitchen)
     (at tray328 kitchen)
     (at_kitchen_bread bread378)
     (at_kitchen_content content343)
     (not_allergic_gluten child255)
     (waiting child255 table383)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child255)
    )
  )
)
