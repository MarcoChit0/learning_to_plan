; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 602416

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child447 - child
    bread372 - bread-portion
    content105 - content-portion
    tray68 tray180 tray449 - tray
    table469 table235 table12 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray68 kitchen)
     (at tray180 kitchen)
     (at tray449 kitchen)
     (at_kitchen_bread bread372)
     (at_kitchen_content content105)
     (not_allergic_gluten child447)
     (waiting child447 table469)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child447)
    )
  )
)
