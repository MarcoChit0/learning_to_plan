; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 711735

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child139 child86 - child
    bread154 bread231 - bread-portion
    content377 content297 - content-portion
    tray464 tray191 - tray
    table422 table108 table156 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray464 kitchen)
     (at tray191 kitchen)
     (at_kitchen_bread bread154)
     (at_kitchen_bread bread231)
     (at_kitchen_content content377)
     (at_kitchen_content content297)
     (not_allergic_gluten child86)
     (not_allergic_gluten child139)
     (waiting child139 table156)
     (waiting child86 table422)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child139)
     (served child86)
    )
  )
)
