; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 307728

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child154 child141 child423 - child
    bread371 bread214 bread84 - bread-portion
    content73 content163 content209 - content-portion
    tray84 tray83 tray401 - tray
    table169 table484 table414 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray84 kitchen)
     (at tray83 kitchen)
     (at tray401 kitchen)
     (at_kitchen_bread bread371)
     (at_kitchen_bread bread214)
     (at_kitchen_bread bread84)
     (at_kitchen_content content73)
     (at_kitchen_content content163)
     (at_kitchen_content content209)
     (not_allergic_gluten child154)
     (not_allergic_gluten child423)
     (not_allergic_gluten child141)
     (waiting child154 table484)
     (waiting child141 table484)
     (waiting child423 table169)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child154)
     (served child141)
     (served child423)
    )
  )
)
