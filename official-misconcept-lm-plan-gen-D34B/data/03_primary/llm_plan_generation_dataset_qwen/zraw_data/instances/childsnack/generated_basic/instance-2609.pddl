; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 405959

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child212 - child
    bread315 - bread-portion
    content276 - content-portion
    tray289 tray212 tray327 - tray
    table152 table147 table446 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray289 kitchen)
     (at tray212 kitchen)
     (at tray327 kitchen)
     (at_kitchen_bread bread315)
     (at_kitchen_content content276)
     (not_allergic_gluten child212)
     (waiting child212 table152)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child212)
    )
  )
)
