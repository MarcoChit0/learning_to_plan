; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 236196

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child245 - child
    bread231 - bread-portion
    content421 - content-portion
    tray68 tray156 tray313 - tray
    table428 table386 table92 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray68 kitchen)
     (at tray156 kitchen)
     (at tray313 kitchen)
     (at_kitchen_bread bread231)
     (at_kitchen_content content421)
     (not_allergic_gluten child245)
     (waiting child245 table92)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child245)
    )
  )
)
